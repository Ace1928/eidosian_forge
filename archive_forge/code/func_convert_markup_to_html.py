import pyparsing as pp
def convert_markup_to_html(opening, closing):

    def conversionParseAction(s, l, t):
        return opening + wiki_markup.transformString(t[1][1:-1]) + closing
    return conversionParseAction