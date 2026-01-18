from openstack import resource
def comma_separated_list(value):
    if value is None:
        return None
    else:
        return ','.join(value)