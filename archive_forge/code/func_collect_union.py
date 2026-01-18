def collect_union(type_list, base, nodes):
    for t in type_list:
        if is_node(t['name']):
            nodes.append(base)
        elif is_shape(t['name']):
            nodes = collect_nodes(t['value'], base, nodes)
        elif t['name'] == 'arrayOf':
            nodes = collect_array(t['value'], base, nodes)
        elif t['name'] == 'objectOf':
            nodes = collect_object(t['value'], base, nodes)
    return nodes