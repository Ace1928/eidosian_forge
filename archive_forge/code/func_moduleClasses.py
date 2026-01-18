import sys
import glob
import inspect
def moduleClasses(mod):

    def P(obj, m=mod.__name__, CT=type):
        return type(obj) == CT and obj.__module__ == m
    try:
        return inspect.getmembers(mod, P)[0][1]
    except:
        return None