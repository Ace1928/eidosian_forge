import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def interactive_demo(trace=False):
    import random
    import sys
    HELP = '\n    1-%d: Select the corresponding feature structure\n    q: Quit\n    t: Turn tracing on or off\n    l: List all feature structures\n    ?: Help\n    '
    print('\n    This demo will repeatedly present you with a list of feature\n    structures, and ask you to choose two for unification.  Whenever a\n    new feature structure is generated, it is added to the list of\n    choices that you can pick from.  However, since this can be a\n    large number of feature structures, the demo will only print out a\n    random subset for you to choose between at a given time.  If you\n    want to see the complete lists, type "l".  For a list of valid\n    commands, type "?".\n    ')
    print('Press "Enter" to continue...')
    sys.stdin.readline()
    fstruct_strings = ['[agr=[number=sing, gender=masc]]', '[agr=[gender=masc, person=3]]', '[agr=[gender=fem, person=3]]', '[subj=[agr=(1)[]], agr->(1)]', '[obj=?x]', '[subj=?x]', '[/=None]', '[/=NP]', '[cat=NP]', '[cat=VP]', '[cat=PP]', '[subj=[agr=[gender=?y]], obj=[agr=[gender=?y]]]', '[gender=masc, agr=?C]', '[gender=?S, agr=[gender=?S,person=3]]']
    all_fstructs = [(i, FeatStruct(fstruct_strings[i])) for i in range(len(fstruct_strings))]

    def list_fstructs(fstructs):
        for i, fstruct in fstructs:
            print()
            lines = ('%s' % fstruct).split('\n')
            print('%3d: %s' % (i + 1, lines[0]))
            for line in lines[1:]:
                print('     ' + line)
        print()
    while True:
        MAX_CHOICES = 5
        if len(all_fstructs) > MAX_CHOICES:
            fstructs = sorted(random.sample(all_fstructs, MAX_CHOICES))
        else:
            fstructs = all_fstructs
        print('_' * 75)
        print('Choose two feature structures to unify:')
        list_fstructs(fstructs)
        selected = [None, None]
        for nth, i in (('First', 0), ('Second', 1)):
            while selected[i] is None:
                print('%s feature structure (1-%d,q,t,l,?): ' % (nth, len(all_fstructs)), end=' ')
                try:
                    input = sys.stdin.readline().strip()
                    if input in ('q', 'Q', 'x', 'X'):
                        return
                    if input in ('t', 'T'):
                        trace = not trace
                        print('   Trace = %s' % trace)
                        continue
                    if input in ('h', 'H', '?'):
                        print(HELP % len(fstructs))
                        continue
                    if input in ('l', 'L'):
                        list_fstructs(all_fstructs)
                        continue
                    num = int(input) - 1
                    selected[i] = all_fstructs[num][1]
                    print()
                except:
                    print('Bad sentence number')
                    continue
        if trace:
            result = selected[0].unify(selected[1], trace=1)
        else:
            result = display_unification(selected[0], selected[1])
        if result is not None:
            for i, fstruct in all_fstructs:
                if repr(result) == repr(fstruct):
                    break
            else:
                all_fstructs.append((len(all_fstructs), result))
        print('\nType "Enter" to continue unifying; or "q" to quit.')
        input = sys.stdin.readline().strip()
        if input in ('q', 'Q', 'x', 'X'):
            return