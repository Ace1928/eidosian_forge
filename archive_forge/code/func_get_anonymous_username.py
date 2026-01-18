import importlib
import random
import re
import warnings
def get_anonymous_username() -> str:
    """
    Get a random user-name based on the moons of Jupyter.
    This function returns names like "Anonymous Io" or "Anonymous Metis".
    """
    return moons_of_jupyter[random.randint(0, len(moons_of_jupyter) - 1)]