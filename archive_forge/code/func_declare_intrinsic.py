import collections
from llvmlite.ir import context, values, types, _utils
def declare_intrinsic(self, intrinsic, tys=(), fnty=None):

    def _error():
        raise NotImplementedError('unknown intrinsic %r with %d types' % (intrinsic, len(tys)))
    if intrinsic in {'llvm.cttz', 'llvm.ctlz', 'llvm.fma'}:
        suffixes = [tys[0].intrinsic_name]
    else:
        suffixes = [t.intrinsic_name for t in tys]
    name = '.'.join([intrinsic] + suffixes)
    if name in self.globals:
        return self.globals[name]
    if fnty is not None:
        pass
    elif len(tys) == 0 and intrinsic == 'llvm.assume':
        fnty = types.FunctionType(types.VoidType(), [types.IntType(1)])
    elif len(tys) == 1:
        if intrinsic == 'llvm.powi':
            fnty = types.FunctionType(tys[0], [tys[0], types.IntType(32)])
        elif intrinsic == 'llvm.pow':
            fnty = types.FunctionType(tys[0], tys * 2)
        elif intrinsic == 'llvm.convert.from.fp16':
            fnty = types.FunctionType(tys[0], [types.IntType(16)])
        elif intrinsic == 'llvm.convert.to.fp16':
            fnty = types.FunctionType(types.IntType(16), tys)
        else:
            fnty = types.FunctionType(tys[0], tys)
    elif len(tys) == 2:
        if intrinsic == 'llvm.memset':
            tys = [tys[0], types.IntType(8), tys[1], types.IntType(1)]
            fnty = types.FunctionType(types.VoidType(), tys)
        elif intrinsic in {'llvm.cttz', 'llvm.ctlz'}:
            tys = [tys[0], types.IntType(1)]
            fnty = types.FunctionType(tys[0], tys)
        else:
            _error()
    elif len(tys) == 3:
        if intrinsic in ('llvm.memcpy', 'llvm.memmove'):
            tys = tys + [types.IntType(1)]
            fnty = types.FunctionType(types.VoidType(), tys)
        elif intrinsic == 'llvm.fma':
            tys = [tys[0]] * 3
            fnty = types.FunctionType(tys[0], tys)
        else:
            _error()
    else:
        _error()
    return values.Function(self, fnty, name=name)