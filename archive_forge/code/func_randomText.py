from reportlab import rl_config
def randomText(theme=STARTUP, sentences=5):
    if type(theme) == type(''):
        if theme.lower() == 'chomsky':
            return chomsky(sentences)
        elif theme.upper() in ('STARTUP', 'COMPUTERS', 'BLAH', 'BUZZWORD', 'STARTREK', 'PRINTING', 'PYTHON'):
            theme = globals()[theme.upper()]
        else:
            raise ValueError('Unknown theme "%s"' % theme)
    from random import randint, choice
    RANDOMWORDS = theme
    output = ''
    for sentenceno in range(randint(1, sentences)):
        output = output + 'Blah'
        for wordno in range(randint(10, 25)):
            if randint(0, 4) == 0:
                word = choice(RANDOMWORDS)
            else:
                word = 'blah'
            output = output + ' ' + word
        output = output + '. '
    return output