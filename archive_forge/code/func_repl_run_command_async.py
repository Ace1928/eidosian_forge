import asyncio
import errno
import signal
from pexpect import EOF
@asyncio.coroutine
def repl_run_command_async(repl, cmdlines, timeout=-1):
    res = []
    repl.child.sendline(cmdlines[0])
    for line in cmdlines[1:]:
        yield from repl._expect_prompt(timeout=timeout, async_=True)
        res.append(repl.child.before)
        repl.child.sendline(line)
    prompt_idx = (yield from repl._expect_prompt(timeout=timeout, async_=True))
    if prompt_idx == 1:
        repl.child.kill(signal.SIGINT)
        yield from repl._expect_prompt(timeout=1, async_=True)
        raise ValueError('Continuation prompt found - input was incomplete:')
    return ''.join(res + [repl.child.before])