from __future__ import print_function
def extract_common_symbols(symbols1, symbols2, already_extracted):
    for symbol1, lineno1, line1, hexcode1 in symbols1:
        for symbol2, lineno2, line2, hexcode2 in symbols2:
            if symbol1 in already_extracted or symbol2 in already_extracted:
                continue
            if symbol1 == symbol2 + 'f':
                print('// Different Name; Redefine')
                print(line2)
                print('#define %s %s' % (symbol1, symbol2))
            elif symbol1 == symbol2:
                already_extracted.append(symbol1)
                print(line1)
                if symbol1 == 'GLclampf;':
                    print('typedef GLclampf GLclampd;')
            elif hexcode1 and hexcode2 and (hexcode1 == hexcode2):
                already_extracted.append(symbol1)
                already_extracted.append(symbol2)
                print('// Different Name; Redefine')
                print(line2)
                print('#define %s %s' % (symbol1, symbol2))