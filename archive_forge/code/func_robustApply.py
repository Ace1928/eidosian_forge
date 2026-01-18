import sys
def robustApply(receiver, *arguments, **named):
    """Call receiver with arguments and an appropriate subset of named
    """
    receiver, codeObject, startIndex = function(receiver)
    acceptable = codeObject.co_varnames[startIndex + len(arguments):codeObject.co_argcount]
    for name in codeObject.co_varnames[startIndex:startIndex + len(arguments)]:
        if name in named:
            raise TypeError('Argument %r specified both positionally and as a keyword for calling %r' % (name, receiver))
    if not codeObject.co_flags & 8:
        named = dict([(k, v) for k, v in named.items() if k in acceptable])
    return receiver(*arguments, **named)