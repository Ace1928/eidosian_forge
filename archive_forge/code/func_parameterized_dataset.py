import functools
import types
def parameterized_dataset(build_data):
    """Simple decorator to mark a test method for processing."""

    def decorator(func):
        func.__dict__['build_data'] = build_data
        return func
    return decorator