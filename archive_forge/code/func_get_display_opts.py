import functools
import re
import sys
from Xlib.support import lock
def get_display_opts(options, argv=sys.argv):
    """display, name, db, args = get_display_opts(options, [argv])

    Parse X OPTIONS from ARGV (or sys.argv if not provided).

    Connect to the display specified by a *.display resource if one is
    set, or to the default X display otherwise.  Extract the
    RESOURCE_MANAGER property and insert all resources from ARGV.

    The four return values are:
      DISPLAY -- the display object
      NAME    -- the application name (the filname of ARGV[0])
      DB      -- the created resource database
      ARGS    -- any remaining arguments
    """
    from Xlib import display, Xatom
    import os
    name = os.path.splitext(os.path.basename(argv[0]))[0]
    optdb = ResourceDB()
    leftargv = optdb.getopt(name, argv[1:], options)
    dname = optdb.get(name + '.display', name + '.Display', None)
    d = display.Display(dname)
    rdbstring = d.screen(0).root.get_full_property(Xatom.RESOURCE_MANAGER, Xatom.STRING)
    if rdbstring:
        data = rdbstring.value
    else:
        data = None
    db = ResourceDB(string=data)
    db.update(optdb)
    return (d, name, db, leftargv)