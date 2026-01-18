from pycadf import cadftype
def is_valid_resource(value):
    for type in RESOURCE_TAXONOMY:
        if value.startswith(type):
            return True
    return False