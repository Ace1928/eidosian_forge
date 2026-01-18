import paste.util.threadinglocal as threadinglocal
def enable_restoration(self, stacked):
    """Replace the specified StackedObjectProxy's methods with their
        respective restoration versions.

        _current_obj_restoration forces recovery of the saved proxied object
        when a restoration context is active in the current thread.

        _push/pop_object_restoration avoid pushing/popping data
        (pushing/popping is only done at the Registry level) when a restoration
        context is active in the current thread"""
    if '_current_obj_orig' in stacked.__dict__:
        return
    for func_name in ('_current_obj', '_push_object', '_pop_object'):
        orig_func = getattr(stacked, func_name)
        restoration_func = getattr(stacked, func_name + '_restoration')
        stacked.__dict__[func_name + '_orig'] = orig_func
        stacked.__dict__[func_name] = restoration_func