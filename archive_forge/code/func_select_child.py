import re
def select_child(result):
    for elem in result:
        for e in elem.iter():
            if e is not elem:
                yield e