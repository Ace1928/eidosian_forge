def filter_by_name(record, parent, length):
    name = record['name']
    if name is None:
        return False
    return (name + '.')[:length] == parent