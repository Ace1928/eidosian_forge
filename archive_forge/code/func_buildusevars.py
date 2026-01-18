from .auxfuncs import (
def buildusevars(m, r):
    ret = {}
    outmess('\t\tBuilding use variable hooks for module "%s" (feature only for F90/F95)...\n' % m['name'])
    varsmap = {}
    revmap = {}
    if 'map' in r:
        for k in r['map'].keys():
            if r['map'][k] in revmap:
                outmess('\t\t\tVariable "%s<=%s" is already mapped by "%s". Skipping.\n' % (r['map'][k], k, revmap[r['map'][k]]))
            else:
                revmap[r['map'][k]] = k
    if 'only' in r and r['only']:
        for v in r['map'].keys():
            if r['map'][v] in m['vars']:
                if revmap[r['map'][v]] == v:
                    varsmap[v] = r['map'][v]
                else:
                    outmess('\t\t\tIgnoring map "%s=>%s". See above.\n' % (v, r['map'][v]))
            else:
                outmess('\t\t\tNo definition for variable "%s=>%s". Skipping.\n' % (v, r['map'][v]))
    else:
        for v in m['vars'].keys():
            if v in revmap:
                varsmap[v] = revmap[v]
            else:
                varsmap[v] = v
    for v in varsmap.keys():
        ret = dictappend(ret, buildusevar(v, varsmap[v], m['vars'], m['name']))
    return ret