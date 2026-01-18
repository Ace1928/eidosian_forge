import re, sys, os, tempfile, json
def remove_forbidden(poly_str):
    """
    PHCpack doesn't allow variables with {i, I, e, E} in the name.
    Also, with the phcpy interface, it uses "m" for the multiplicity
    and "t" for the homotopy parameter.
    """
    for bad, replacement in replacements:
        poly_str = poly_str.replace(bad, replacement)
    return poly_str