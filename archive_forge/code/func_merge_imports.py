import deap
from copy import copy
def merge_imports(old_dict, new_dict):
    for key in new_dict.keys():
        if key in old_dict.keys():
            old_dict[key] = set(old_dict[key]) | set(new_dict[key])
        else:
            old_dict[key] = set(new_dict[key])