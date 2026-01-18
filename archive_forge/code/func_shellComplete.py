import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
def shellComplete(config, cmdName, words, shellCompFile):
    """
    Perform shell completion.

    A completion function (shell script) is generated for the requested
    shell and written to C{shellCompFile}, typically C{stdout}. The result
    is then eval'd by the shell to produce the desired completions.

    @type config: L{twisted.python.usage.Options}
    @param config: The L{twisted.python.usage.Options} instance to generate
        completions for.

    @type cmdName: C{str}
    @param cmdName: The name of the command we're generating completions for.
        In the case of zsh, this is used to print an appropriate
        "#compdef $CMD" line at the top of the output. This is
        not necessary for the functionality of the system, but it
        helps in debugging, since the output we produce is properly
        formed and may be saved in a file and used as a stand-alone
        completion function.

    @type words: C{list} of C{str}
    @param words: The raw command-line words passed to use by the shell
        stub function. argv[0] has already been stripped off.

    @type shellCompFile: C{file}
    @param shellCompFile: The file to write completion data to.
    """
    if shellCompFile and ioType(shellCompFile) == str:
        shellCompFile = shellCompFile.buffer
    shellName, position = words[-1].split(':')
    position = int(position)
    position -= 2
    cWord = words[position]
    while position >= 1:
        if words[position - 1].startswith('-'):
            position -= 1
        else:
            break
    words = words[:position]
    subCommands = getattr(config, 'subCommands', None)
    if subCommands:
        args = None
        try:
            opts, args = getopt.getopt(words, config.shortOpt, config.longOpt)
        except getopt.error:
            pass
        if args:
            for cmd, short, parser, doc in config.subCommands:
                if args[0] == cmd or args[0] == short:
                    subOptions = parser()
                    subOptions.parent = config
                    gen: ZshBuilder = ZshSubcommandBuilder(subOptions, config, cmdName, shellCompFile)
                    gen.write()
                    return
        genSubs = True
        if cWord.startswith('-'):
            genSubs = False
        gen = ZshBuilder(config, cmdName, shellCompFile)
        gen.write(genSubs=genSubs)
    else:
        gen = ZshBuilder(config, cmdName, shellCompFile)
        gen.write()