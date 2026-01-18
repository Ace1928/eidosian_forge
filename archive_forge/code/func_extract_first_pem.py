from __future__ import absolute_import, division, print_function
def extract_first_pem(text):
    """
    Given one PEM or multiple concatenated PEM objects, return only the first one, or None if there is none.
    """
    all_pems = split_pem_list(text)
    if not all_pems:
        return None
    return all_pems[0]