import time
def list2cols_with_rename(names_and_keys, objs):
    cols = [i[0] for i in names_and_keys]
    keys = [i[1] for i in names_and_keys]
    return (cols, [tuple([o.get(k, '') for k in keys]) for o in objs])