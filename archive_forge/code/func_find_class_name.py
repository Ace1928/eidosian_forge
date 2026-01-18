import sys
def find_class_name(val):
    class_name = str(val.__class__)
    if class_name.find('.') != -1:
        class_name = class_name.split('.')[-1]
    elif class_name.find("'") != -1:
        class_name = class_name[class_name.index("'") + 1:]
    if class_name.endswith("'>"):
        class_name = class_name[:-2]
    return class_name