from urllib import parse
@staticmethod
def get_dimensions_url_string(dimensions):
    dim_list = list()
    for k, v in dimensions.items():
        if isinstance(v, (list, tuple)):
            v = v[-1]
        if v:
            dim_str = k + ':' + v
        else:
            dim_str = k
        dim_list.append(dim_str)
    return ','.join(dim_list)