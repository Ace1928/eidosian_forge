import sys
from enchant.checker import SpellChecker
def read_command(self):
    cmd = input('>> ')
    cmd = cmd.strip()
    if cmd.isdigit():
        repl = int(cmd)
        suggs = self.error.suggest()
        if repl >= len(suggs):
            print([warning('No suggestion number'), repl])
            return False
        print([success("Replacing '%s' with '%s'" % (color(self.error.word, color='red'), color(suggs[repl], color='green')))])
        self.error.replace(suggs[repl])
        return True
    if cmd[0] == 'R':
        if not cmd[1:].isdigit():
            print([warning("Badly formatted command (try 'help')")])
            return False
        repl = int(cmd[1:])
        suggs = self.error.suggest()
        if repl >= len(suggs):
            print([warning('No suggestion number'), repl])
            return False
        self.error.replace_always(suggs[repl])
        return True
    if cmd == 'i':
        return True
    if cmd == 'I':
        self.error.ignore_always()
        return True
    if cmd == 'a':
        self.error.add()
        return True
    if cmd == 'e':
        repl = input(info('New Word: '))
        self.error.replace(repl.strip())
        return True
    if cmd == 'q':
        self._stop = True
        return True
    if 'help'.startswith(cmd.lower()):
        self.print_help()
        return False
    print([warning("Badly formatted command (try 'help')")])
    return False