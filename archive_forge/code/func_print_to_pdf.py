from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def print_to_pdf(landscape: typing.Optional[bool]=None, display_header_footer: typing.Optional[bool]=None, print_background: typing.Optional[bool]=None, scale: typing.Optional[float]=None, paper_width: typing.Optional[float]=None, paper_height: typing.Optional[float]=None, margin_top: typing.Optional[float]=None, margin_bottom: typing.Optional[float]=None, margin_left: typing.Optional[float]=None, margin_right: typing.Optional[float]=None, page_ranges: typing.Optional[str]=None, header_template: typing.Optional[str]=None, footer_template: typing.Optional[str]=None, prefer_css_page_size: typing.Optional[bool]=None, transfer_mode: typing.Optional[str]=None, generate_tagged_pdf: typing.Optional[bool]=None, generate_document_outline: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[str, typing.Optional[io.StreamHandle]]]:
    """
    Print page as PDF.

    :param landscape: *(Optional)* Paper orientation. Defaults to false.
    :param display_header_footer: *(Optional)* Display header and footer. Defaults to false.
    :param print_background: *(Optional)* Print background graphics. Defaults to false.
    :param scale: *(Optional)* Scale of the webpage rendering. Defaults to 1.
    :param paper_width: *(Optional)* Paper width in inches. Defaults to 8.5 inches.
    :param paper_height: *(Optional)* Paper height in inches. Defaults to 11 inches.
    :param margin_top: *(Optional)* Top margin in inches. Defaults to 1cm (~0.4 inches).
    :param margin_bottom: *(Optional)* Bottom margin in inches. Defaults to 1cm (~0.4 inches).
    :param margin_left: *(Optional)* Left margin in inches. Defaults to 1cm (~0.4 inches).
    :param margin_right: *(Optional)* Right margin in inches. Defaults to 1cm (~0.4 inches).
    :param page_ranges: *(Optional)* Paper ranges to print, one based, e.g., '1-5, 8, 11-13'. Pages are printed in the document order, not in the order specified, and no more than once. Defaults to empty string, which implies the entire document is printed. The page numbers are quietly capped to actual page count of the document, and ranges beyond the end of the document are ignored. If this results in no pages to print, an error is reported. It is an error to specify a range with start greater than end.
    :param header_template: *(Optional)* HTML template for the print header. Should be valid HTML markup with following classes used to inject printing values into them: - ```date````: formatted print date - ````title````: document title - ````url````: document location - ````pageNumber````: current page number - ````totalPages````: total pages in the document  For example, ````<span class=title></span>```` would generate span containing the title.
    :param footer_template: *(Optional)* HTML template for the print footer. Should use the same format as the ````headerTemplate````.
    :param prefer_css_page_size: *(Optional)* Whether or not to prefer page size as defined by css. Defaults to false, in which case the content will be scaled to fit the paper size.
    :param transfer_mode: **(EXPERIMENTAL)** *(Optional)* return as stream
    :param generate_tagged_pdf: **(EXPERIMENTAL)** *(Optional)* Whether or not to generate tagged (accessible) PDF. Defaults to embedder choice.
    :param generate_document_outline: **(EXPERIMENTAL)** *(Optional)* Whether or not to embed the document outline into the PDF.
    :returns: A tuple with the following items:

        0. **data** - Base64-encoded pdf data. Empty if `` returnAsStream` is specified.
        1. **stream** - *(Optional)* A handle of the stream that holds resulting PDF data.
    """
    params: T_JSON_DICT = dict()
    if landscape is not None:
        params['landscape'] = landscape
    if display_header_footer is not None:
        params['displayHeaderFooter'] = display_header_footer
    if print_background is not None:
        params['printBackground'] = print_background
    if scale is not None:
        params['scale'] = scale
    if paper_width is not None:
        params['paperWidth'] = paper_width
    if paper_height is not None:
        params['paperHeight'] = paper_height
    if margin_top is not None:
        params['marginTop'] = margin_top
    if margin_bottom is not None:
        params['marginBottom'] = margin_bottom
    if margin_left is not None:
        params['marginLeft'] = margin_left
    if margin_right is not None:
        params['marginRight'] = margin_right
    if page_ranges is not None:
        params['pageRanges'] = page_ranges
    if header_template is not None:
        params['headerTemplate'] = header_template
    if footer_template is not None:
        params['footerTemplate'] = footer_template
    if prefer_css_page_size is not None:
        params['preferCSSPageSize'] = prefer_css_page_size
    if transfer_mode is not None:
        params['transferMode'] = transfer_mode
    if generate_tagged_pdf is not None:
        params['generateTaggedPDF'] = generate_tagged_pdf
    if generate_document_outline is not None:
        params['generateDocumentOutline'] = generate_document_outline
    cmd_dict: T_JSON_DICT = {'method': 'Page.printToPDF', 'params': params}
    json = (yield cmd_dict)
    return (str(json['data']), io.StreamHandle.from_json(json['stream']) if 'stream' in json else None)