from contextlib import contextmanager
from jedi import debug
from jedi.inference.base_value import NO_VALUES
def execution_recursion_decorator(default=NO_VALUES):

    def decorator(func):

        def wrapper(self, **kwargs):
            detector = self.inference_state.execution_recursion_detector
            limit_reached = detector.push_execution(self)
            try:
                if limit_reached:
                    result = default
                else:
                    result = func(self, **kwargs)
            finally:
                detector.pop_execution()
            return result
        return wrapper
    return decorator