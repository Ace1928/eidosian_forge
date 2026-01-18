from __future__ import annotations
from io import (
import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
def test_incorrect_xsl_apply(geom_df):
    lxml_etree = pytest.importorskip('lxml.etree')
    xsl = '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n    <xsl:output method="xml" encoding="utf-8" indent="yes" />\n    <xsl:strip-space elements="*"/>\n\n    <xsl:template match="@*|node()">\n        <xsl:copy>\n            <xsl:copy-of select="document(\'non_existent.xml\')/*"/>\n        </xsl:copy>\n    </xsl:template>\n</xsl:stylesheet>'
    with pytest.raises(lxml_etree.XSLTApplyError, match='Cannot resolve URI'):
        with tm.ensure_clean('test.xml') as path:
            geom_df.to_xml(path, stylesheet=xsl)