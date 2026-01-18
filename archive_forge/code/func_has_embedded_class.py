import re
def has_embedded_class(classes):
    return any((_mf2_e_properties_re.match(c) for c in classes))