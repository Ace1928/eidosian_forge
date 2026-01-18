import time
import logging
import array
from weakref import ref as weakref_ref
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import value
from pyomo.core.expr.numvalue import is_fixed, ZeroConstant
from pyomo.core.base.set_types import Any
from pyomo.core.base import SortComponents, Var
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.constraint import (
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.repn import generate_standard_repn
from collections.abc import Mapping
def compile_block_linear_constraints(parent_block, constraint_name, skip_trivial_constraints=False, single_precision_storage=False, verbose=False, descend_into=True):
    if verbose:
        print('')
        print('Compiling linear constraints on block with name: %s' % parent_block.name)
    if not parent_block.is_constructed():
        raise RuntimeError("Attempting to compile block '%s' with unconstructed component(s)" % parent_block.name)
    SparseMat_pRows = []
    SparseMat_jCols = []
    SparseMat_Vals = []
    Ranges = []
    RangeTypes = []

    def _get_bound(exp):
        if exp is None:
            return None
        if is_fixed(exp):
            return value(exp)
        raise ValueError('non-fixed bound: ' + str(exp))
    start_time = time.time()
    if verbose:
        print('Sorting active blocks...')
    sortOrder = SortComponents.indices | SortComponents.alphabetical
    all_blocks = [_b for _b in parent_block.block_data_objects(active=True, sort=sortOrder, descend_into=descend_into)]
    stop_time = time.time()
    if verbose:
        print('Time to sort active blocks: %.2f seconds' % (stop_time - start_time))
    start_time = time.time()
    if verbose:
        print('Collecting variables on active blocks...')
    VarSymbolToVarObject = []
    for block in all_blocks:
        VarSymbolToVarObject.extend(block.component_data_objects(Var, sort=sortOrder, descend_into=False))
    VarIDToVarSymbol = dict(((id(vardata), index) for index, vardata in enumerate(VarSymbolToVarObject)))
    stop_time = time.time()
    if verbose:
        print('Time to collect variables on active blocks: %.2f seconds' % (stop_time - start_time))
    start_time = time.time()
    if verbose:
        print('Compiling active linear constraints...')
    constraint_data_to_remove = []
    empty_constraint_containers_to_remove = []
    constraint_containers_to_remove = []
    constraint_containers_to_check = set()
    referenced_variable_symbols = set()
    nnz = 0
    nrows = 0
    SparseMat_pRows = [0]
    for block in all_blocks:
        if hasattr(block, '_repn'):
            del block._repn
        for constraint in block.component_objects(Constraint, active=True, sort=sortOrder, descend_into=False):
            assert not isinstance(constraint, MatrixConstraint)
            if len(constraint) == 0:
                empty_constraint_containers_to_remove.append((block, constraint))
            else:
                singleton = isinstance(constraint, ScalarConstraint)
                for index, constraint_data in list(constraint.items()):
                    if constraint_data.body.__class__ in native_numeric_types or constraint_data.body.polynomial_degree() <= 1:
                        if singleton:
                            constraint_containers_to_remove.append((block, constraint))
                        else:
                            constraint_data_to_remove.append((constraint, index))
                            constraint_containers_to_check.add((block, constraint))
                        repn = generate_standard_repn(constraint_data.body)
                        assert repn.nonlinear_expr is None
                        row_variable_symbols = []
                        row_coefficients = []
                        if len(repn.linear_vars) == 0:
                            if skip_trivial_constraints:
                                continue
                        else:
                            row_variable_symbols = [VarIDToVarSymbol[id(vardata)] for vardata in repn.linear_vars]
                            referenced_variable_symbols.update(row_variable_symbols)
                            assert repn.linear_coefs is not None
                            row_coefficients = repn.linear_coefs
                        SparseMat_pRows.append(SparseMat_pRows[-1] + len(row_variable_symbols))
                        SparseMat_jCols.extend(row_variable_symbols)
                        SparseMat_Vals.extend(row_coefficients)
                        nnz += len(row_variable_symbols)
                        nrows += 1
                        L = _get_bound(constraint_data.lower)
                        U = _get_bound(constraint_data.upper)
                        constant = value(repn.constant)
                        Ranges.append(L - constant if L is not None else 0)
                        Ranges.append(U - constant if U is not None else 0)
                        if L is not None and U is not None and (not constraint_data.equality):
                            RangeTypes.append(MatrixConstraint.LowerBound | MatrixConstraint.UpperBound)
                        elif constraint_data.equality:
                            RangeTypes.append(MatrixConstraint.Equality)
                        elif L is not None:
                            assert U is None
                            RangeTypes.append(MatrixConstraint.LowerBound)
                        else:
                            assert U is not None
                            RangeTypes.append(MatrixConstraint.UpperBound)
                        constraint[index] = Constraint.Skip
    ncols = len(referenced_variable_symbols)
    stop_time = time.time()
    if verbose:
        print('Time to compile active linear constraints: %.2f seconds' % (stop_time - start_time))
    start_time = time.time()
    if verbose:
        print('Removing compiled constraint objects...')
    constraints_removed = 0
    constraint_containers_removed = 0
    for block, constraint in empty_constraint_containers_to_remove:
        block.del_component(constraint)
        constraint_containers_removed += 1
    for constraint, index in constraint_data_to_remove:
        constraints_removed += 1
    for block, constraint in constraint_containers_to_remove:
        block.del_component(constraint)
        constraints_removed += 1
        constraint_containers_removed += 1
    for block, constraint in constraint_containers_to_check:
        if len(constraint) == 0:
            block.del_component(constraint)
            constraint_containers_removed += 1
    stop_time = time.time()
    if verbose:
        print('Eliminated %s constraints and %s Constraint container objects' % (constraints_removed, constraint_containers_removed))
        print('Time to remove compiled constraint objects: %.2f seconds' % (stop_time - start_time))
    start_time = time.time()
    if verbose:
        print('Assigning variable column indices...')
    ColumnIndexToVarSymbol = sorted(referenced_variable_symbols)
    VarSymbolToColumnIndex = dict(((symbol, column) for column, symbol in enumerate(ColumnIndexToVarSymbol)))
    SparseMat_jCols = [VarSymbolToColumnIndex[symbol] for symbol in SparseMat_jCols]
    del VarSymbolToColumnIndex
    ColumnIndexToVarObject = [VarSymbolToVarObject[var_symbol] for var_symbol in ColumnIndexToVarSymbol]
    stop_time = time.time()
    if verbose:
        print('Time to assign variable column indices: %.2f seconds' % (stop_time - start_time))
    start_time = time.time()
    if verbose:
        print('Converting compiled constraint data to array storage...')
        print('  - Using %s precision for numeric values' % ('single' if single_precision_storage else 'double'))
    number_storage = 'f' if single_precision_storage else 'd'
    SparseMat_pRows = array.array('L', SparseMat_pRows)
    SparseMat_jCols = array.array('L', SparseMat_jCols)
    SparseMat_Vals = array.array(number_storage, SparseMat_Vals)
    Ranges = array.array(number_storage, Ranges)
    RangeTypes = array.array('B', RangeTypes)
    stop_time = time.time()
    if verbose:
        storage_bytes = SparseMat_pRows.buffer_info()[1] * SparseMat_pRows.itemsize + SparseMat_jCols.buffer_info()[1] * SparseMat_jCols.itemsize + SparseMat_Vals.buffer_info()[1] * SparseMat_Vals.itemsize + Ranges.buffer_info()[1] * Ranges.itemsize + RangeTypes.buffer_info()[1] * RangeTypes.itemsize
        print('Sparse Matrix Dimension:')
        print('  - Rows: ' + str(nrows))
        print('  - Cols: ' + str(ncols))
        print('  - Nonzeros: ' + str(nnz))
        print('Compiled Data Storage: ' + str(_label_bytes(storage_bytes)))
        print('Time to convert compiled constraint data to array storage: %.2f seconds' % (stop_time - start_time))
    parent_block.add_component(constraint_name, MatrixConstraint(nrows, ncols, nnz, SparseMat_pRows, SparseMat_jCols, SparseMat_Vals, Ranges, RangeTypes, ColumnIndexToVarObject))