import torch
from typing import Any, Set, Dict, List, Tuple, OrderedDict
from collections import OrderedDict as OrdDict
def generate_histogram_visualization(self, feature_filter: str, module_fqn_filter: str='', num_bins: int=10):
    """
        Takes in a feature and optional module_filter and plots the histogram of desired data.

        Note:
            Only features in the report that have tensor value data can be viewed as a histogram
            If you want to plot a histogram from all the channel values of a specific feature for
                a specific model, make sure to specify both the model and the feature properly
                in the filters and you should be able to see a distribution of the channel data

        Args:
            feature_filter (str, optional): Filters the features presented to only those that
                contain this filter substring
                Default = "", results in all the features being printed
            module_fqn_filter (str, optional): Only includes modules that contains this string
                Default = "", results in all the modules in the reports to be visible in the table
            num_bins (int, optional): The number of bins to create the histogram with
                Default = 10, the values will be split into 10 equal sized bins

        Example Use:
            >>> # xdoctest: +SKIP
            >>> mod_report_visualizer.generategenerate_histogram_visualization_plot_visualization(
            ...     feature_filter = "per_channel_min",
            ...     module_fqn_filter = "block1"
            ... )
            # outputs histogram of per_channel_min information for all modules in block1 of model
                information is gathered across all channels for all modules in block 1 for the
                per_channel_min and is displayed in a histogram of equally sized bins
        """
    if not got_matplotlib:
        print('make sure to install matplotlib and try again.')
        return None
    x_data, y_data, data_per_channel = self._get_plottable_data(feature_filter, module_fqn_filter)
    ax = plt.subplot()
    ax.set_xlabel(feature_filter)
    ax.set_ylabel('Frequency')
    ax.set_title(feature_filter + ' Histogram')
    if data_per_channel:
        all_data = []
        for channel_info in y_data:
            all_data.extend(channel_info)
        val, bins, _ = plt.hist(all_data, bins=num_bins, stacked=True, rwidth=0.8)
        plt.xticks(bins)
    else:
        val, bins, _ = plt.hist(y_data, bins=num_bins, stacked=False, rwidth=0.8)
        plt.xticks(bins)
    plt.show()