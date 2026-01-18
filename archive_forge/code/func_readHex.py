import ast
import string
def readHex(content):
    """Convert a list of hex strings to binary data."""
    return deHexStr(strjoin((chunk for chunk in content if isinstance(chunk, str))))