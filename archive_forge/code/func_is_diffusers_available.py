import importlib
import sys
def is_diffusers_available() -> bool:
    return importlib.util.find_spec('diffusers') is not None