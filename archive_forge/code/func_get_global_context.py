from llvmlite.binding import ffi
def get_global_context():
    return GlobalContextRef(ffi.lib.LLVMPY_GetGlobalContext())