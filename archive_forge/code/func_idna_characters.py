from __future__ import absolute_import
def idna_characters():
    """
        Returns a string containing IDNA characters.
        """
    global _idnaCharacters
    if not _idnaCharacters:
        result = []
        dataFileName = join(dirname(__file__), 'idna-tables-properties.csv.gz')
        with open_gzip(dataFileName) as dataFile:
            reader = csv_reader((line.decode('utf-8') for line in dataFile), delimiter=',')
            next(reader)
            for row in reader:
                codes, prop, description = row
                if prop != 'PVALID':
                    continue
                startEnd = row[0].split('-', 1)
                if len(startEnd) == 1:
                    startEnd.append(startEnd[0])
                start, end = (int(i, 16) for i in startEnd)
                for i in range(start, end + 1):
                    if i > maxunicode:
                        break
                    result.append(unichr(i))
        _idnaCharacters = u''.join(result)
    return _idnaCharacters