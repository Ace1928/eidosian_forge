from io import open
from pathlib import Path
def find_package_file(*path):
    """Return the full path to a file from the itables package"""
    current_path = Path(__file__).parent
    return Path(current_path, *path)