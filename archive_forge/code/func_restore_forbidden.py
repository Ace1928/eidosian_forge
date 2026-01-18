import re, sys, os, tempfile, json
def restore_forbidden(var_str):
    for bad, replacement in replacements:
        var_str = var_str.replace(replacement, bad)
    return var_str