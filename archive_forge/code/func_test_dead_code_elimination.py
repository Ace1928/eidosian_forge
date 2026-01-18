import numba
from numba.tests.support import TestCase, unittest
from numba.core.registry import cpu_target
from numba.core.compiler import CompilerBase, Flags
from numba.core.compiler_machinery import PassManager
from numba.core import types, ir, bytecode, compiler, ir_utils, registry
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference,
from numba.experimental import jitclass
def test_dead_code_elimination(self):

    class Tester(CompilerBase):

        @classmethod
        def mk_pipeline(cls, args, return_type=None, flags=None, locals={}, library=None, typing_context=None, target_context=None):
            if not flags:
                flags = Flags()
            flags.nrt = True
            if typing_context is None:
                typing_context = registry.cpu_target.typing_context
            if target_context is None:
                target_context = registry.cpu_target.target_context
            return cls(typing_context, target_context, library, args, return_type, flags, locals)

        def compile_to_ir(self, func, DCE=False):
            """
                Compile and return IR
                """
            func_id = bytecode.FunctionIdentity.from_function(func)
            self.state.func_id = func_id
            ExtractByteCode().run_pass(self.state)
            state = self.state
            name = 'DCE_testing'
            pm = PassManager(name)
            pm.add_pass(TranslateByteCode, 'analyzing bytecode')
            pm.add_pass(FixupArgs, 'fix up args')
            pm.add_pass(IRProcessing, 'processing IR')
            pm.add_pass(NopythonTypeInference, 'nopython frontend')
            if DCE is True:
                pm.add_pass(DeadCodeElimination, 'DCE after typing')
            pm.finalize()
            pm.run(state)
            return state.func_ir

    def check_initial_ir(the_ir):
        self.assertEqual(len(the_ir.blocks), 1)
        block = the_ir.blocks[0]
        deads = []
        for x in block.find_insts(ir.Assign):
            if isinstance(getattr(x, 'target', None), ir.Var):
                if 'dead' in getattr(x.target, 'name', ''):
                    deads.append(x)
        self.assertEqual(len(deads), 2)
        for d in deads:
            const_val = the_ir.get_definition(d.value)
            self.assertTrue(int('0x%s' % d.target.name, 16), const_val.value)
        return deads

    def check_dce_ir(the_ir):
        self.assertEqual(len(the_ir.blocks), 1)
        block = the_ir.blocks[0]
        deads = []
        consts = []
        for x in block.find_insts(ir.Assign):
            if isinstance(getattr(x, 'target', None), ir.Var):
                if 'dead' in getattr(x.target, 'name', ''):
                    deads.append(x)
            if isinstance(getattr(x, 'value', None), ir.Const):
                consts.append(x)
        self.assertEqual(len(deads), 0)
        for x in consts:
            self.assertTrue(x.value.value not in [57005, 3735936685])

    def foo(x):
        y = x + 1
        dead = 57005
        z = y + 2
        deaddead = 3735936685
        ret = z * z
        return ret
    test_pipeline = Tester.mk_pipeline((types.intp,))
    no_dce = test_pipeline.compile_to_ir(foo)
    removed = check_initial_ir(no_dce)
    test_pipeline = Tester.mk_pipeline((types.intp,))
    w_dce = test_pipeline.compile_to_ir(foo, DCE=True)
    check_dce_ir(w_dce)
    self.assertEqual(len(no_dce.blocks[0].body) - len(removed), len(w_dce.blocks[0].body))