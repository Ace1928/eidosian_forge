from plotly import exceptions, optional_imports
from plotly.graph_objs import graph_objs
def get_table_font_color(self):
    """
        Fill font-color array.

        Table text color can vary by row so this extends a single color or
        creates an array to set a header color and two alternating colors to
        create the striped table pattern.

        :rtype (list[list]) all_font_colors: list of font colors for each row
            in table.
        """
    if len(self.font_colors) == 1:
        all_font_colors = self.font_colors * len(self.table_text)
    elif len(self.font_colors) == 3:
        all_font_colors = list(range(len(self.table_text)))
        all_font_colors[0] = self.font_colors[0]
        for i in range(1, len(self.table_text), 2):
            all_font_colors[i] = self.font_colors[1]
        for i in range(2, len(self.table_text), 2):
            all_font_colors[i] = self.font_colors[2]
    elif len(self.font_colors) == len(self.table_text):
        all_font_colors = self.font_colors
    else:
        all_font_colors = ['#000000'] * len(self.table_text)
    return all_font_colors