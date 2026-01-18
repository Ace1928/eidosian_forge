from pyparsing import *
def convertToHTML_A(s, l, t):
    try:
        text, url = t[0].split('->')
    except ValueError:
        raise ParseFatalException(s, l, 'invalid URL link reference: ' + t[0])
    return '<A href="{0}">{1}</A>'.format(url, text)