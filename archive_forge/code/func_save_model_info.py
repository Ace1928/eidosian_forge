import csv
import gzip
import json
import math
import optparse
import os
import pickle
import re
import sys
from pickle import Unpickler
import numpy as np
import requests
from pylab import *
from scipy import interp, stats
from sklearn import cross_validation, metrics, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, make_scorer, precision_score, recall_score,
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, SDWriter
from rdkit.ML.Descriptors import MoleculeDescriptors
from the one dimensional weights.
def save_model_info(self, outfile, mode='html'):
    """create html- or csv-File for models according to mode (default: "html")"""
    if mode == 'csv':
        if not outfile.endswith('.csv'):
            outfile += '.csv'
        csv_file = open(outfile, 'wb')
        csv_file_writer = csv.writer(csv_file, delimiter=';', quotechar=' ')
        for line in self.csv_text:
            csv_file_writer.writerow(line)
        csv_file.flush()
        csv_file.close()
    elif mode == 'html':
        if not outfile.endswith('.html'):
            outfile += '.html'

        def lines2list(lines):
            return lines

        def list2html(data, act, inact):
            html_head = '<html>\n<head>\n<meta http-equiv="Content-Type" content="text/html; charset=utf-8">\n<title></title>\n<style type="text/css">\ntable {\n  max-width: 100%;\n  background-color: transparent;\n}\n\nth {\n  text-align: left;\n}\n\n.table {\n  width: 100%;\n  margin-bottom: 20px;\n}\n\n.table > thead > tr > th,\n.table > tbody > tr > th,\n.table > tfoot > tr > th,\n.table > thead > tr > td,\n.table > tbody > tr > td,\n.table > tfoot > tr > td {\n  padding: 8px;\n  line-height: 1.428571429;\n  vertical-align: top;\n  border-top: 1px solid #dddddd;\n}\n\n.table > thead > tr > th {MSC1013123\n  vertical-align: bottom;\n  border-bottom: 2px solid #dddddd;\n}\n\n.table > caption + thead > tr:first-child > th,\n.table > colgroup + thead > tr:first-child > th,\n.table > thead:first-child > tr:first-child > th,\n.table > caption + thead > tr:first-child > td,\n.table > colgroup + thead > tr:first-child > td,\n.table > thead:first-child > tr:first-child > td {\n  border-top: 0;\n}\n\n.table > tbody + tbody {\n  border-top: 2px solid #dddddd;\n}\n\n.table .table {\n  background-color: #ffffff;\n}\n\n.table-condensed > thead > tr > th,\n.table-condensed > tbody > tr > th,\n.table-condensed > tfoot > tr > th,\n.table-condensed > thead > tr > td,\n.table-condensed > tbody > tr > td,\n.table-condensed > tfoot > tr > td {\n  padding: 5px;\n}\n\n.table-bordered {\n  border: 1px solid #dddddd;\n}\n\n.table-bordered > thead > tr > th,\n.table-bordered > tbody > tr > th,\n.table-bordered > tfoot > tr > th,\n.table-bordered > thead > tr > td,\n.table-bordered > tbody > tr > td,\n.table-bordered > tfoot > tr > td {\n  border: 1px solid #dddddd;\n}\n\n.table-bordered > thead > tr > th,\n.table-bordered > thead > tr > td {\n  border-bottom-width: 2px;\n}\n\n.table-striped > tbody > tr:nth-child(odd) > td,\n.table-striped > tbody > tr:nth-child(odd) > th {\n  background-color: #f9f9f9;\n}\n\n.table-hover > tbody > tr:hover > td,\n.table-hover > tbody > tr:hover > th {\n  background-color: #f5f5f5;\n}\n\ntable col[class*="col-"] {\n  position: static;\n  display: table-column;\n  float: none;\n}\n\ntable td[class*="col-"],\ntable th[class*="col-"] {\n  display: table-cell;\n  float: none;\n}\n\n.table > thead > tr > .active,\n.table > tbody > tr > .active,\n.table > tfoot > tr > .active,\n.table > thead > .active > td,\n.table > tbody > .active > td,\n.table > tfoot > .active > td,\n.table > thead > .active > th,\n.table > tbody > .active > th,\n.table > tfoot > .active > th {\n  background-color: #f5f5f5;\n}\n\n.table-hover > tbody > tr > .active:hover,\n.table-hover > tbody > .active:hover > td,\n.table-hover > tbody > .active:hover > th {\n  background-color: #e8e8e8;\n}\n\n.table > thead > tr > .success,\n.table > tbody > tr > .success,\n.table > tfoot > tr > .success,\n.table > thead > .success > td,\n.table > tbody > .success > td,\n.table > tfoot > .success > td,\n.table > thead > .success > th,\n.table > tbody > .success > th,\n.table > tfoot > .success > th {\n  background-color: #dff0d8;\n}\n\n.table-hover > tbody > tr > .success:hover,\n.table-hover > tbody > .success:hover > td,\n.table-hover > tbody > .success:hover > th {\n  background-color: #d0e9c6;\n}\n\n.table > thead > tr > .danger,\n.table > tbody > tr > .danger,\n.table > tfoot > tr > .danger,\n.table > thead > .danger > td,\n.table > tbody > .danger > td,\n.table > tfoot > .danger > td,\n.table > thead > .danger > th,\n.table > tbody > .danger > th,\n.table > tfoot > .danger > th {\n  background-color: #f2dede;\n}\n\n.table-hover > tbody > tr > .danger:hover,\n.table-hover > tbody > .danger:hover > td,\n.table-hover > tbody > .danger:hover > th {\n  background-color: #ebcccc;\n}\n\n.table > thead > tr > .warning,\n.table > tbody > tr > .warning,\n.table > tfoot > tr > .warning,\n.table > thead > .warning > td,\n.table > tbody > .warning > td,\n.table > tfoot > .warning > td,\n.table > thead > .warning > th,\n.table > tbody > .warning > th,\n.table > tfoot > .warning > th {\n  background-color: #fcf8e3;\n}\n\n.table-hover > tbody > tr > .warning:hover,\n.table-hover > tbody > .warning:hover > td,\n.table-hover > tbody > .warning:hover > th {\n  background-color: #faf2cc;\n}\n\n@media (max-width: 767px) {\n  .table-responsive {\n    width: 100%;\n    margin-bottom: 15px;\n    overflow-x: scroll;\n    overflow-y: hidden;\n    border: 1px solid #dddddd;\n    -ms-overflow-style: -ms-autohiding-scrollbar;\n    -webkit-overflow-scrolling: touch;\n  }\n  .table-responsive > .table {\n    margin-bottom: 0;\n  }\n  .table-responsive > .table > thead > tr > th,\n  .table-responsive > .table > tbody > tr > th,\n  .table-responsive > .table > tfoot > tr > th,\n  .table-responsive > .table > thead > tr > td,\n  .table-responsive > .table > tbody > tr > td,\n  .table-responsive > .table > tfoot > tr > td {\n    white-space: nowrap;\n  }\n  .table-responsive > .table-bordered {\n    border: 0;\n  }\n  .table-responsive > .table-bordered > thead > tr > th:first-child,\n  .table-responsive > .table-bordered > tbody > tr > th:first-child,\n  .table-responsive > .table-bordered > tfoot > tr > th:first-child,\n  .table-responsive > .table-bordered > thead > tr > td:first-child,\n  .table-responsive > .table-bordered > tbody > tr > td:first-child,\n  .table-responsive > .table-bordered > tfoot > tr > td:first-child {\n    border-left: 0;\n  }\n  .table-responsive > .table-bordered > thead > tr > th:last-child,\n  .table-responsive > .table-bordered > tbody > tr > th:last-child,\n  .table-responsive > .table-bordered > tfoot > tr > th:last-child,\n  .table-responsive > .table-bordered > thead > tr > td:last-child,\n  .table-responsive > .table-bordered > tbody > tr > td:last-child,\n  .table-responsive > .table-bordered > tfoot > tr > td:last-child {\n    border-right: 0;\n  }\n  .table-responsive > .table-bordered > tbody > tr:last-child > th,\n  .table-responsive > .table-bordered > tfoot > tr:last-child > th,\n  .table-responsive > .table-bordered > tbody > tr:last-child > td,\n  .table-responsive > .table-bordered > tfoot > tr:last-child > td {\n    border-bottom: 0;\n  }\n}\n</style>\n</head>\n<body>\n<p style="padding-left:10px;padding-top:10px;font-size:200&#37;">Data for Models</p>\n<p style="padding-left:10px;padding-right:10px;">'
            html_topPlot_start = '<table style="vertical-align:top; background-color=#CCCCCC">\n<tr align="left" valign="top"><td><img src="pieplot.png"></td><td><H3>Distribution</H3><font color="#00C000">active %d</font><br><font color="#FF0000">inactive %d</td><td>'
            html_topPlot_bottom = '</td></tr></table>'
            html_tableStart = '<table class="table table-bordered table-condensed">\n<thead>\n<tr>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n</tr>\n</thead>\n<tbody>'
            html_tElements = '\n<tr bgcolor = "%s">\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td><a href="%s">model.pkl</a></td>\n</tr>'
            html_bottomPlot = '</tbody>\n</table>\n<img src="barplot.png"><br>'
            html_foot = '\n</p>\n</body>\n</html>'
            html_kappa_table_head = '<table class="table table-bordered table-condensed">\n<thead>\n<tr>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n<th>%s</th>\n</tr>\n</thead>\n<tbody>'
            html_kappa_table_element = '<tr bgcolor = "%s">\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td>%s</td>\n<td><a href="%s">model.pkl</a></td>\n</tr>'
            html_kappa_table_bottom = '</tbody>\n</table>\n<img src="barplot.png"><br>'
            best, worst = findBestWorst(data)
            html = []
            html.append(html_head)
            html.append(html_topPlot_start % (act, inact))
            html.append(html_topPlot_bottom)
            html.append(html_tableStart % tuple(data[0]))
            i = 0
            for l in data[1:len(data)]:
                l_replaced = []
                for elem in l:
                    elem_string = str(elem)
                    if elem_string.find('pkl') == -1:
                        l_replaced.append(elem_string.replace('_', '±'))
                    else:
                        l_replaced.append(elem_string)
                c = ''
                if i == best:
                    c = '#9CC089'
                if i == worst:
                    c = '#FF3333'
                html.append(html_tElements % tuple([c] + l_replaced))
                i += 1
            html.append(html_bottomPlot)
            html.append(html_foot)
            createBarPlot(data)
            return html

        def writeHtml(html, outf):
            outf_h = open(outf, 'w')
            for block in html:
                outf_h.write(block)
            outf_h.flush()
            outf_h.close()
            return

        def findBestWorst(data):
            auc = [float(x[6].split('_')[0]) for x in data[1:]]
            max_index, min_index = (auc.index(max(auc)), auc.index(min(auc)))
            return (max_index, min_index)

        def createPiePlot(cpds):

            def getActInact(cpds):
                act, inact = (0, 0)
                for cpd in cpds:
                    if int(cpd.GetProp('TL')) == 0:
                        inact += 1
                    else:
                        act += 1
                return (act, inact)
            act_count, inact_count = getActInact(cpds)
            print("act/inact from TL's %d/%d" % (act_count, inact_count))
            fig = plt.figure(figsize=(2, 2))
            pie = plt.pie([inact_count, act_count], colors=('r', 'g'))
            fig.savefig('pieplot.png', transparent=True)
            return (act_count, inact_count)

        def createBarPlot(data):

            def getLists(data, col):
                accList = []
                errList = []
                for x in data[1:]:
                    if x[col].find('_') == -1:
                        continue
                    if x[col].find('.pkl') != -1:
                        continue
                    spl = x[col].split('_')
                    accList.append(float(spl[0]))
                    errList.append(float(spl[1]))
                return (accList, errList)

            def plotLists(cnt):
                result = []
                clr = ['#DD1E2F', '#EBB035', '#06A2CB', '#218559', '#D0C6B1', '#192823', '#DDAACC']
                for i in range(1, cnt):
                    list, errList = getLists(data, i)
                    result.append(ax.bar(ticks + width * i, list, width, color=clr[i - 1], yerr=errList))
                return result
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 6)
            ticks = np.arange(0.0, 12.0, 1.2)
            if len(self.model) == 1:
                ticks = np.arange(0.0, 1.0, 1.5)
            width = 0.15
            plots = plotLists(8)
            ax.set_xticks(ticks + 0.75)
            ax.set_xticklabels([str(x) for x in range(1, 11, 1)])
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('# model')
            ax.set_xlim(-0.3, 14)
            ax.set_ylim(-0.1, 1.2)
            ax.legend(tuple(plots), [x for x in data[0][1:8]], 'upper right')
            best, worst = findBestWorst(data)
            if len(self.model) > 1:
                ax.annotate('best', xy=(ticks[best], 0.85), xytext=(ticks[best] + 0.25, 1.1), color='green')
                ax.annotate('worst', xy=(ticks[worst], 0.85), xytext=(ticks[worst] + 0.25, 1.1), color='red')
            fig.savefig('barplot.png', transparent=True)
            return
        act, inact = createPiePlot(self.sd_entries)
        lines = self.csv_text
        data = lines2list(lines)
        html = list2html(data, act, inact)
        writeHtml(html, outfile)
    return True