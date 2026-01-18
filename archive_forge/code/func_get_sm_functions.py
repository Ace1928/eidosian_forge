from __future__ import print_function
def get_sm_functions():
    f = opener.open('http://docs.sourcemod.net/api/SMfuncs.js')
    r = re.compile('SMfunctions\\[\\d+\\] = Array \\("(?:public )?([^,]+)",".+"\\);')
    functions = []
    for line in f:
        m = r.match(line)
        if m is not None:
            functions.append(m.groups()[0])
    return functions