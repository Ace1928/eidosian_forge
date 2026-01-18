from reportlab.lib import colors
from Bio.Graphics.BasicChromosome import ChromosomeSegment
from Bio.Graphics.BasicChromosome import TelomereSegment
def fill_chromosome(self, chromosome):
    """Add the collected segment information to a chromosome for drawing.

        Arguments:
         - chromosome - A Chromosome graphics object that we can add
           chromosome segments to.

        This creates ChromosomeSegment (and TelomereSegment) objects to
        fill in the chromosome. The information is derived from the
        label and count information, with counts transformed to the
        specified color map.

        Returns the chromosome with all of the segments added.
        """
    for seg_num in range(len(self._names)):
        is_end_segment = 0
        if seg_num == 0:
            cur_segment = TelomereSegment()
            is_end_segment = 1
        elif seg_num == len(self._names) - 1:
            cur_segment = TelomereSegment(1)
            is_end_segment = 1
        else:
            cur_segment = ChromosomeSegment()
        seg_name = self._names[seg_num]
        if self._count_info[seg_name] > 0:
            color = self._color_from_count(self._count_info[seg_name])
            cur_segment.fill_color = color
        if self._label_info[seg_name] is not None:
            cur_segment.label = self._label_info[seg_name]
        if is_end_segment:
            cur_segment.scale = 3
        else:
            cur_segment.scale = self._scale_info[seg_name]
        chromosome.add(cur_segment)
    return chromosome