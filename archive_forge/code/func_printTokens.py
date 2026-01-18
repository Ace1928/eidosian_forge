import os, sys, time, random
def printTokens(self, tokens):
    for i in tokens:
        s = self.idx2token[i]
        try:
            s = s.decode('utf-8')
        except:
            pass
        print(f'{repr(s)}{i}', end=' ')
    print()