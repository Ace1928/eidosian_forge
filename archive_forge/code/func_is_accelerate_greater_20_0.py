import importlib
import sys
def is_accelerate_greater_20_0() -> bool:
    if _is_python_greater_3_8:
        from importlib.metadata import version
        accelerate_version = version('accelerate')
    else:
        import pkg_resources
        accelerate_version = pkg_resources.get_distribution('accelerate').version
    return accelerate_version >= '0.20.0'