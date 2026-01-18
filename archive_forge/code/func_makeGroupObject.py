from pyparsing import *
import_stmt: import_name | import_from
import_name: 'import' dotted_as_names
import_from: 'from' dotted_name 'import' ('*' | '(' import_as_names ')' | import_as_names)
import_as_name: NAME [NAME NAME]
import_as_names: import_as_name (',' import_as_name)* [',']
def makeGroupObject(cls):

    def groupAction(s, l, t):
        try:
            return cls(t[0].asList())
        except Exception:
            return cls(t)
    return groupAction