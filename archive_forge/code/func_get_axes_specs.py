from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode
def get_axes_specs(env, axes):
    """
    get_axes_specs(env, axes) -> list of (access, packing) specs for each axis.
    access is one of 'full', 'ptr' or 'direct'
    packing is one of 'contig', 'strided' or 'follow'
    """
    cythonscope = env.global_scope().context.cython_scope
    cythonscope.load_cythonscope()
    viewscope = cythonscope.viewscope
    access_specs = tuple([viewscope.lookup(name) for name in ('full', 'direct', 'ptr')])
    packing_specs = tuple([viewscope.lookup(name) for name in ('contig', 'strided', 'follow')])
    is_f_contig, is_c_contig = (False, False)
    default_access, default_packing = ('direct', 'strided')
    cf_access, cf_packing = (default_access, 'follow')
    axes_specs = []
    for idx, axis in enumerate(axes):
        if not axis.start.is_none:
            raise CompileError(axis.start.pos, START_ERR)
        if not axis.stop.is_none:
            raise CompileError(axis.stop.pos, STOP_ERR)
        if axis.step.is_none:
            axes_specs.append((default_access, default_packing))
        elif isinstance(axis.step, IntNode):
            if axis.step.compile_time_value(env) != 1:
                raise CompileError(axis.step.pos, STEP_ERR)
            axes_specs.append((cf_access, 'cfcontig'))
        elif isinstance(axis.step, (NameNode, AttributeNode)):
            entry = _get_resolved_spec(env, axis.step)
            if entry.name in view_constant_to_access_packing:
                axes_specs.append(view_constant_to_access_packing[entry.name])
            else:
                raise CompileError(axis.step.pos, INVALID_ERR)
        else:
            raise CompileError(axis.step.pos, INVALID_ERR)
    contig_dim = 0
    is_contig = False
    for idx, (access, packing) in enumerate(axes_specs):
        if packing == 'cfcontig':
            if is_contig:
                raise CompileError(axis.step.pos, BOTH_CF_ERR)
            contig_dim = idx
            axes_specs[idx] = (access, 'contig')
            is_contig = True
    if is_contig:
        if contig_dim == len(axes) - 1:
            is_c_contig = True
        else:
            is_f_contig = True
            if contig_dim and (not axes_specs[contig_dim - 1][0] in ('full', 'ptr')):
                raise CompileError(axes[contig_dim].pos, 'Fortran contiguous specifier must follow an indirect dimension')
        if is_c_contig:
            contig_dim = -1
            for idx, (access, packing) in enumerate(reversed(axes_specs)):
                if access in ('ptr', 'full'):
                    contig_dim = len(axes) - idx - 1
        start = contig_dim + 1
        stop = len(axes) - is_c_contig
        for idx, (access, packing) in enumerate(axes_specs[start:stop]):
            idx = contig_dim + 1 + idx
            if access != 'direct':
                raise CompileError(axes[idx].pos, 'Indirect dimension may not follow Fortran contiguous dimension')
            if packing == 'contig':
                raise CompileError(axes[idx].pos, 'Dimension may not be contiguous')
            axes_specs[idx] = (access, cf_packing)
        if is_c_contig:
            a, p = axes_specs[-1]
            axes_specs[-1] = (a, 'contig')
    validate_axes_specs([axis.start.pos for axis in axes], axes_specs, is_c_contig, is_f_contig)
    return axes_specs