import unittest
@box(IntervalType)
def box_interval(typ, val, c):
    """
            Convert a native interval structure to an Interval object.
            """
    ret_ptr = cgutils.alloca_once(c.builder, c.pyapi.pyobj)
    fail_obj = c.pyapi.get_null_object()
    with ExitStack() as stack:
        interval = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
        lo_obj = c.box(types.float64, interval.lo)
        with cgutils.early_exit_if_null(c.builder, stack, lo_obj):
            c.builder.store(fail_obj, ret_ptr)
        hi_obj = c.box(types.float64, interval.hi)
        with cgutils.early_exit_if_null(c.builder, stack, hi_obj):
            c.pyapi.decref(lo_obj)
            c.builder.store(fail_obj, ret_ptr)
        class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Interval))
        with cgutils.early_exit_if_null(c.builder, stack, class_obj):
            c.pyapi.decref(lo_obj)
            c.pyapi.decref(hi_obj)
            c.builder.store(fail_obj, ret_ptr)
        res = c.pyapi.call_function_objargs(class_obj, (lo_obj, hi_obj))
        c.pyapi.decref(lo_obj)
        c.pyapi.decref(hi_obj)
        c.pyapi.decref(class_obj)
        c.builder.store(res, ret_ptr)
    return c.builder.load(ret_ptr)