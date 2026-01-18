from __future__ import print_function
def get_lua_functions(version):
    f = urlopen('http://www.lua.org/manual/%s/' % version)
    r = re.compile('^<A HREF="manual.html#pdf-(?!lua|LUA)([^:]+)">\\1</A>')
    functions = []
    for line in f:
        m = r.match(line)
        if m is not None:
            functions.append(m.groups()[0])
    return functions