import re
import _markupbase
from html import unescape
def goahead(self, end):
    rawdata = self.rawdata
    i = 0
    n = len(rawdata)
    while i < n:
        if self.convert_charrefs and (not self.cdata_elem):
            j = rawdata.find('<', i)
            if j < 0:
                amppos = rawdata.rfind('&', max(i, n - 34))
                if amppos >= 0 and (not re.compile('[\\s;]').search(rawdata, amppos)):
                    break
                j = n
        else:
            match = self.interesting.search(rawdata, i)
            if match:
                j = match.start()
            else:
                if self.cdata_elem:
                    break
                j = n
        if i < j:
            if self.convert_charrefs and (not self.cdata_elem):
                self.handle_data(unescape(rawdata[i:j]))
            else:
                self.handle_data(rawdata[i:j])
        i = self.updatepos(i, j)
        if i == n:
            break
        startswith = rawdata.startswith
        if startswith('<', i):
            if starttagopen.match(rawdata, i):
                k = self.parse_starttag(i)
            elif startswith('</', i):
                k = self.parse_endtag(i)
            elif startswith('<!--', i):
                k = self.parse_comment(i)
            elif startswith('<?', i):
                k = self.parse_pi(i)
            elif startswith('<!', i):
                k = self.parse_html_declaration(i)
            elif i + 1 < n:
                self.handle_data('<')
                k = i + 1
            else:
                break
            if k < 0:
                if not end:
                    break
                k = rawdata.find('>', i + 1)
                if k < 0:
                    k = rawdata.find('<', i + 1)
                    if k < 0:
                        k = i + 1
                else:
                    k += 1
                if self.convert_charrefs and (not self.cdata_elem):
                    self.handle_data(unescape(rawdata[i:k]))
                else:
                    self.handle_data(rawdata[i:k])
            i = self.updatepos(i, k)
        elif startswith('&#', i):
            match = charref.match(rawdata, i)
            if match:
                name = match.group()[2:-1]
                self.handle_charref(name)
                k = match.end()
                if not startswith(';', k - 1):
                    k = k - 1
                i = self.updatepos(i, k)
                continue
            else:
                if ';' in rawdata[i:]:
                    self.handle_data(rawdata[i:i + 2])
                    i = self.updatepos(i, i + 2)
                break
        elif startswith('&', i):
            match = entityref.match(rawdata, i)
            if match:
                name = match.group(1)
                self.handle_entityref(name)
                k = match.end()
                if not startswith(';', k - 1):
                    k = k - 1
                i = self.updatepos(i, k)
                continue
            match = incomplete.match(rawdata, i)
            if match:
                if end and match.group() == rawdata[i:]:
                    k = match.end()
                    if k <= i:
                        k = n
                    i = self.updatepos(i, i + 1)
                break
            elif i + 1 < n:
                self.handle_data('&')
                i = self.updatepos(i, i + 1)
            else:
                break
        else:
            assert 0, 'interesting.search() lied'
    if end and i < n and (not self.cdata_elem):
        if self.convert_charrefs and (not self.cdata_elem):
            self.handle_data(unescape(rawdata[i:n]))
        else:
            self.handle_data(rawdata[i:n])
        i = self.updatepos(i, n)
    self.rawdata = rawdata[i:]