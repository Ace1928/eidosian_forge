import os
import sys as _sys
def name_to_classname(name: str) -> str:
    words = name.split('_')
    class_name = ''
    for w in words:
        class_name += w[0].upper() + w[1:]
    return class_name