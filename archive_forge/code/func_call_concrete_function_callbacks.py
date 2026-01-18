def call_concrete_function_callbacks(concrete_fn):
    """Calls registered callbacks against new ConcreteFunctions."""
    for callback in CONCRETE_FUNCTION_CALLBACKS:
        callback(concrete_fn)