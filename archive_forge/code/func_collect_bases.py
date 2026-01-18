def collect_bases(curr_class, classes_to_generate, memo=None):
    ret = []
    if memo is None:
        memo = {}
    base_definitions = curr_class['base_definitions']
    for base_definition in base_definitions:
        if base_definition not in memo:
            ret.append(base_definition)
            ret.extend(collect_bases(classes_to_generate[base_definition], classes_to_generate, memo))
    return ret