from .. import core
@core.builtin
def num_threads(_builder=None):
    return core.constexpr(_builder.target.num_warps * 32)