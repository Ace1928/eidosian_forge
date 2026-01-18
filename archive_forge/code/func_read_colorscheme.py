from reportlab.lib import colors
def read_colorscheme(self, filename):
    """Load colour scheme from file.

        Reads information from a file containing color information and stores
        it internally.

        Argument filename is the location of a file defining colors in
        tab-separated format plaintext as::

            INT \\t RED \\t GREEN \\t BLUE \\t Comment

        Where RED, GREEN and BLUE are intensities in the range 0 -> 255, e.g.::

            2 \\t 255 \\t 0 \\t 0 \\t Red: Information transfer

        """
    with open(filename).readlines() as lines:
        for line in lines:
            data = line.strip().split('\t')
            try:
                label = int(data[0])
                red, green, blue = (int(data[1]), int(data[2]), int(data[3]))
                if len(data) > 4:
                    comment = data[4]
                else:
                    comment = ''
                self._colorscheme[label] = (self.int255_color((red, green, blue)), comment)
            except ValueError:
                raise ValueError('Expected INT \t INT \t INT \t INT \t string input') from None