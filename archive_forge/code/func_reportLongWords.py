from pyparsing import *
def reportLongWords(st, locn, toks):
    word = toks[0]
    if len(word) > 3:
        print("Found '%s' on line %d at column %d" % (word, lineno(locn, st), col(locn, st)))
        print('The full line of text was:')
        print("'%s'" % line(locn, st))
        print(' ' * col(locn, st) + ' ^')
        print()