import sys
def shellcomplete_commands(outfile=None):
    """List all commands"""
    from inspect import getdoc
    from . import commands
    commands.install_bzr_command_hooks()
    if outfile is None:
        outfile = sys.stdout
    cmds = []
    for cmdname in commands.all_command_names():
        cmd = commands.get_cmd_object(cmdname)
        cmds.append((cmdname, cmd))
        for alias in cmd.aliases:
            cmds.append((alias, cmd))
    cmds.sort()
    for cmdname, cmd in cmds:
        if cmd.hidden:
            continue
        doc = getdoc(cmd)
        if doc is None:
            outfile.write(cmdname + '\n')
        else:
            doclines = doc.splitlines()
            firstline = doclines[0].lower()
            outfile.write(cmdname + ':' + firstline[0:-1] + '\n')